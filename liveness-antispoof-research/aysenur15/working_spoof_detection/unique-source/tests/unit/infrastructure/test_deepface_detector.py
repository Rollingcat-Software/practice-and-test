"""Unit tests for DeepFace face-detector post-filtering."""

from app.infrastructure.ml.detectors.deepface_detector import DeepFaceDetector


def test_nested_false_positive_filters_mouth_like_candidate_inside_primary_face():
    primary = (200, 120, 220, 240)
    mouth_like = (255, 255, 90, 72)

    assert DeepFaceDetector._is_nested_false_positive(mouth_like, primary) is True


def test_nested_false_positive_keeps_separate_second_face_candidate():
    primary = (200, 120, 220, 240)
    separate_face = (28, 96, 140, 150)

    assert DeepFaceDetector._is_nested_false_positive(separate_face, primary) is False
