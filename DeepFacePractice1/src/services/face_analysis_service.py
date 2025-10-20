"""
Face Analysis Service
Handles analyzing facial attributes (age, gender, emotion, race).

Educational Notes:
- DeepFace can extract multiple attributes from a single image
- Useful for demographics analysis, emotion detection, etc.
- Each attribute uses specialized neural networks
"""

from typing import List, Dict

from deepface import DeepFace

from ..models.face_analysis_result import FaceAnalysisResult


class FaceAnalysisService:
    """
    Service for comprehensive facial analysis.

    Demonstrates SRP: Handles only analysis operations,
    separate from verification and recognition.
    """

    # Attributes that can be analyzed
    AVAILABLE_ACTIONS = [
        "age",  # Age estimation
        "gender",  # Gender classification
        "emotion",  # Emotion recognition (7 emotions)
        "race",  # Ethnicity classification
    ]

    # Emotions that can be detected
    EMOTIONS = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral"
    ]

    # Race categories
    RACES = [
        "asian",
        "indian",
        "black",
        "white",
        "middle eastern",
        "latino hispanic"
    ]

    def __init__(
            self,
            detector_backend: str = "opencv",
            enforce_detection: bool = True
    ):
        """
        Initialize the analysis service.

        Args:
            detector_backend: Which face detection algorithm to use
            enforce_detection: If True, raises error when no face detected
        """
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection

    def analyze(
            self,
            img_path: str,
            actions: List[str] = None
    ) -> FaceAnalysisResult:
        """
        Analyze facial attributes in an image.

        Args:
            img_path: Path to the image
            actions: List of attributes to analyze (default: all)

        Returns:
            FaceAnalysisResult object with all analyzed attributes

        Raises:
            ValueError: If face detection fails or invalid actions specified
        """
        if actions is None:
            actions = self.AVAILABLE_ACTIONS

        # Validate actions
        invalid_actions = set(actions) - set(self.AVAILABLE_ACTIONS)
        if invalid_actions:
            raise ValueError(
                f"Invalid actions: {invalid_actions}. "
                f"Choose from: {', '.join(self.AVAILABLE_ACTIONS)}"
            )

        try:
            # Call DeepFace.analyze
            # Returns a list of results (one per detected face)
            raw_results = DeepFace.analyze(
                img_path=img_path,
                actions=actions,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                silent=True  # Suppress progress bars
            )

            # Take first face (you can extend this to handle multiple faces)
            raw_result = raw_results[0] if isinstance(raw_results, list) else raw_results

            # Extract data safely with .get() to handle missing keys
            age = raw_result.get("age")
            gender_data = raw_result.get("gender", {})
            emotion_data = raw_result.get("emotion", {})
            race_data = raw_result.get("race", {})
            region = raw_result.get("region")

            # Determine dominant gender
            gender = None
            gender_confidence = None
            if gender_data:
                if isinstance(gender_data, dict):
                    # gender_data is like {"Woman": 99.8, "Man": 0.2}
                    gender = max(gender_data, key=gender_data.get)
                    gender_confidence = gender_data[gender]
                else:
                    gender = str(gender_data)

            # Determine dominant emotion
            dominant_emotion = None
            if emotion_data and isinstance(emotion_data, dict):
                dominant_emotion = max(emotion_data, key=emotion_data.get)

            # Determine dominant race
            dominant_race = None
            if race_data and isinstance(race_data, dict):
                dominant_race = max(race_data, key=race_data.get)

            return FaceAnalysisResult(
                age=age,
                gender=gender,
                gender_confidence=gender_confidence,
                dominant_emotion=dominant_emotion,
                emotion_scores=emotion_data if isinstance(emotion_data, dict) else None,
                dominant_race=dominant_race,
                race_scores=race_data if isinstance(race_data, dict) else None,
                region=region,
                img_path=img_path,
                raw_result=raw_result
            )

        except Exception as e:
            raise ValueError(f"Analysis failed for {img_path}: {str(e)}") from e

    def analyze_emotion_only(self, img_path: str) -> FaceAnalysisResult:
        """Convenience method: analyze only emotion."""
        return self.analyze(img_path, actions=["emotion"])

    def analyze_demographics(self, img_path: str) -> FaceAnalysisResult:
        """Convenience method: analyze age, gender, and race."""
        return self.analyze(img_path, actions=["age", "gender", "race"])

    def batch_analyze(
            self,
            img_paths: List[str],
            actions: List[str] = None
    ) -> Dict[str, FaceAnalysisResult]:
        """
        Analyze multiple images.

        Args:
            img_paths: List of image paths
            actions: List of attributes to analyze (default: all)

        Returns:
            Dictionary mapping image paths to their analysis results
        """
        results = {}

        for img_path in img_paths:
            try:
                result = self.analyze(img_path, actions)
                results[img_path] = result
            except Exception as e:
                print(f"Warning: Failed to analyze {img_path}: {e}")
                continue

        return results
