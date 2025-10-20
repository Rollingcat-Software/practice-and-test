"""
Face Analysis Result Model
Represents comprehensive facial analysis (age, gender, emotion, race).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class FaceAnalysisResult:
    """
    Encapsulates comprehensive face analysis results.

    Educational Notes:
    DeepFace can analyze multiple attributes:
    - Age: Estimated age in years
    - Gender: Male/Female with confidence scores
    - Emotion: angry, disgust, fear, happy, sad, surprise, neutral
    - Race: asian, indian, black, white, middle eastern, latino hispanic
    """
    age: Optional[float]
    gender: Optional[str]
    gender_confidence: Optional[float]
    dominant_emotion: Optional[str]
    emotion_scores: Optional[Dict[str, float]]
    dominant_race: Optional[str]
    race_scores: Optional[Dict[str, float]]
    region: Optional[Dict[str, int]]  # Face bounding box
    img_path: str
    raw_result: Dict[str, Any]

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [f"Face Analysis for: {self.img_path}"]

        if self.age is not None:
            lines.append(f"  Age: ~{self.age:.0f} years")

        if self.gender:
            lines.append(f"  Gender: {self.gender} ({self.gender_confidence:.1f}%)")

        if self.dominant_emotion:
            lines.append(f"  Emotion: {self.dominant_emotion}")
            if self.emotion_scores:
                top_emotions = sorted(
                    self.emotion_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                emotion_str = ", ".join([f"{e}: {s:.1f}%" for e, s in top_emotions])
                lines.append(f"    Top 3: {emotion_str}")

        if self.dominant_race:
            lines.append(f"  Race: {self.dominant_race}")

        if self.region:
            lines.append(f"  Face Location: x={self.region['x']}, y={self.region['y']}, "
                         f"w={self.region['w']}, h={self.region['h']}")

        return "\n".join(lines)
