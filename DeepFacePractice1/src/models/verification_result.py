"""
Verification Result Model
Represents the result of face verification (comparing two faces).
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class VerificationResult:
    """
    Encapsulates face verification results.

    Educational Notes:
    - 'verified': Boolean indicating if faces match
    - 'distance': Similarity metric (lower = more similar)
    - 'threshold': Decision boundary for the model
    - 'model': Which face recognition model was used
    - 'detector': Which face detection algorithm was used
    - 'similarity_metric': How distance was calculated (cosine, euclidean, etc.)
    """
    verified: bool
    distance: float
    threshold: float
    model: str
    detector_backend: str
    similarity_metric: str
    img1_path: str
    img2_path: str
    raw_result: Dict[str, Any]

    @property
    def confidence_percentage(self) -> float:
        """
        Calculate confidence as a percentage.
        The closer distance is to 0, the higher the confidence.
        """
        if self.distance >= self.threshold:
            # Not verified - low confidence
            return max(0, (1 - (self.distance / self.threshold)) * 100)
        else:
            # Verified - high confidence
            return min(100, (1 - (self.distance / self.threshold)) * 100)

    def __str__(self) -> str:
        """Human-readable representation."""
        status = "✓ MATCH" if self.verified else "✗ NO MATCH"
        return (
            f"{status}\n"
            f"  Distance: {self.distance:.4f} (Threshold: {self.threshold:.4f})\n"
            f"  Confidence: {self.confidence_percentage:.2f}%\n"
            f"  Model: {self.model} | Detector: {self.detector_backend}"
        )
