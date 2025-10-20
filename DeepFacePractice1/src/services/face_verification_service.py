"""
Face Verification Service
Handles comparing two faces to determine if they belong to the same person.

Educational Notes:
- Verification = 1:1 comparison (is this person the same as that person?)
- Different from Recognition = 1:N comparison (who is this person from a database?)
"""

from typing import Optional

from deepface import DeepFace

from ..models.verification_result import VerificationResult


class FaceVerificationService:
    """
    Service for face verification operations.

    This class demonstrates the Single Responsibility Principle (SRP):
    - Only handles verification logic
    - Separates concerns from other face operations
    """

    # Available models in DeepFace (from fastest to most accurate)
    AVAILABLE_MODELS = [
        "VGG-Face",  # 2622-D vector, accurate but slow
        "Facenet",  # 128-D vector, good balance
        "Facenet512",  # 512-D vector, more accurate than Facenet
        "OpenFace",  # 128-D vector, fast but less accurate
        "DeepFace",  # 4096-D vector, very accurate
        "DeepID",  # 160-D vector, good for verification
        "ArcFace",  # 512-D vector, state-of-the-art accuracy
        "Dlib",  # 128-D vector, reliable
        "SFace",  # 128-D vector, fast and lightweight
    ]

    # Available detection backends
    AVAILABLE_DETECTORS = [
        "opencv",  # Fastest, less accurate
        "ssd",  # Good balance
        "dlib",  # Accurate, slower
        "mtcnn",  # Very accurate, slow
        "retinaface",  # Most accurate, slowest
        "mediapipe",  # Fast, Google's solution
        "yolov8",  # Latest YOLO, very fast
        "yunet",  # Fast and accurate
        "fastmtcnn",  # Optimized MTCNN
    ]

    # Similarity metrics (distance calculation methods)
    SIMILARITY_METRICS = [
        "cosine",  # Most common, angle-based
        "euclidean",  # Straight-line distance
        "euclidean_l2",  # Normalized Euclidean
    ]

    def __init__(
            self,
            model_name: str = "Facenet512",
            detector_backend: str = "opencv",
            distance_metric: str = "cosine"
    ):
        """
        Initialize the verification service.

        Args:
            model_name: Which face recognition model to use
            detector_backend: Which face detection algorithm to use
            distance_metric: How to calculate similarity
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate that selected parameters are supported."""
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{self.model_name}' not supported. "
                f"Choose from: {', '.join(self.AVAILABLE_MODELS)}"
            )

        if self.detector_backend not in self.AVAILABLE_DETECTORS:
            raise ValueError(
                f"Detector '{self.detector_backend}' not supported. "
                f"Choose from: {', '.join(self.AVAILABLE_DETECTORS)}"
            )

        if self.distance_metric not in self.SIMILARITY_METRICS:
            raise ValueError(
                f"Metric '{self.distance_metric}' not supported. "
                f"Choose from: {', '.join(self.SIMILARITY_METRICS)}"
            )

    def verify(
            self,
            img1_path: str,
            img2_path: str,
            enforce_detection: bool = True
    ) -> VerificationResult:
        """
        Verify if two images contain the same person.

        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            enforce_detection: If True, raises error when no face detected

        Returns:
            VerificationResult object with comparison details

        Raises:
            ValueError: If face detection fails and enforce_detection=True
        """
        try:
            # Call DeepFace.verify
            raw_result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=enforce_detection
            )

            # Wrap in our custom result object
            return VerificationResult(
                verified=raw_result["verified"],
                distance=raw_result["distance"],
                threshold=raw_result["threshold"],
                model=raw_result["model"],
                detector_backend=raw_result["detector_backend"],
                similarity_metric=raw_result["similarity_metric"],
                img1_path=img1_path,
                img2_path=img2_path,
                raw_result=raw_result
            )

        except Exception as e:
            raise ValueError(f"Verification failed: {str(e)}") from e

    def compare_multiple(
            self,
            reference_img: str,
            comparison_imgs: list[str],
            enforce_detection: bool = True
    ) -> list[VerificationResult]:
        """
        Compare one reference image against multiple images.

        Useful for: Finding which image matches a reference photo.

        Args:
            reference_img: The reference image path
            comparison_imgs: List of image paths to compare against
            enforce_detection: If True, raises error when no face detected

        Returns:
            List of VerificationResult objects, one for each comparison
        """
        results = []

        for img_path in comparison_imgs:
            try:
                result = self.verify(reference_img, img_path, enforce_detection)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to compare with {img_path}: {e}")
                # Continue with other images even if one fails
                continue

        return results

    def find_best_match(
            self,
            reference_img: str,
            comparison_imgs: list[str],
            enforce_detection: bool = True
    ) -> Optional[VerificationResult]:
        """
        Find the best matching image from a list.

        Returns the image with the smallest distance (most similar).

        Args:
            reference_img: The reference image path
            comparison_imgs: List of image paths to compare against
            enforce_detection: If True, raises error when no face detected

        Returns:
            VerificationResult for the best match, or None if no matches
        """
        results = self.compare_multiple(
            reference_img,
            comparison_imgs,
            enforce_detection
        )

        if not results:
            return None

        # Find result with minimum distance (most similar)
        return min(results, key=lambda r: r.distance)
