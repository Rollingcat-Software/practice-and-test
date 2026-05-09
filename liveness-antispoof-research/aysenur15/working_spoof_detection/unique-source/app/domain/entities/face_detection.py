"""Face detection result entity."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FaceDetectionResult:
    """Result of face detection operation.

    This is an immutable value object representing the outcome of face detection.
    Following Single Responsibility Principle - only contains detection data.

    Attributes:
        found: Whether a face was detected
        bounding_box: Face bounding box as (x, y, width, height), None if not found
        landmarks: Facial landmarks as numpy array, None if not available
        confidence: Detection confidence score (0.0-1.0)
        antispoof_score: Optional detector anti-spoof score (0.0-1.0)
        antispoof_label: Optional anti-spoof label such as "real" or "spoof"

    Note:
        This class is immutable (frozen) to ensure data integrity.
    """

    found: bool
    bounding_box: Optional[Tuple[int, int, int, int]]
    landmarks: Optional[np.ndarray]
    confidence: float
    antispoof_score: Optional[float] = None
    antispoof_label: Optional[str] = None
    additional_bounding_boxes: tuple[Tuple[int, int, int, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate face detection result data."""
        if self.found and self.bounding_box is None:
            raise ValueError("Bounding box required when face is found")

        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

        if self.antispoof_score is not None and not 0.0 <= self.antispoof_score <= 1.0:
            raise ValueError(f"Anti-spoof score must be 0-1, got {self.antispoof_score}")

        if self.found and self.bounding_box:
            x, y, w, h = self.bounding_box
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid bounding box dimensions: {w}x{h}")

        for bbox in self.additional_bounding_boxes:
            _, _, w, h = bbox
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid additional bounding box dimensions: {w}x{h}")

    def get_face_region(self, image: np.ndarray) -> np.ndarray:
        """Extract face region from image using bounding box.

        Args:
            image: Source image as numpy array (H, W, C)

        Returns:
            Cropped face region

        Raises:
            ValueError: If no bounding box available
        """
        if self.bounding_box is None:
            raise ValueError("No bounding box available")

        x, y, w, h = self.bounding_box
        return image[y : y + h, x : x + w]

    def get_face_center(self) -> Optional[Tuple[int, int]]:
        """Get center point of detected face.

        Returns:
            (x, y) tuple of face center, None if no bounding box
        """
        if self.bounding_box is None:
            return None

        x, y, w, h = self.bounding_box
        return (x + w // 2, y + h // 2)
