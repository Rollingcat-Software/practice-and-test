import numpy as np

from app.domain.entities.face_landmarks import Landmark, LandmarkResult
from app.infrastructure.ml.liveness.face_usability_gate import FaceUsabilityGate


def _face_frame(
    *,
    cover_mouth: bool = False,
    cover_nose: bool = False,
    cover_eyes: bool = False,
) -> np.ndarray:
    frame = np.full((160, 160, 3), (92, 126, 168), dtype=np.uint8)
    frame[20:140, 20:140] = (122, 162, 201)
    frame[30:130:5, 30:130:6] = (138, 176, 214)
    frame[56:74, 50:66] = (35, 50, 70)
    frame[56:74, 94:110] = (35, 50, 70)
    frame[72:100, 76:84] = (80, 105, 130)
    frame[104:118, 56:104] = (44, 58, 84)
    frame[86:134:6, 34:126:7] = (150, 187, 223)
    if cover_eyes:
        frame[52:80, 44:116] = (40, 40, 40)
    if cover_nose:
        frame[68:102, 70:90] = (40, 40, 40)
    if cover_mouth:
        frame[100:140, 40:120] = (40, 40, 40)
    return frame


def _hallucinated_landmarks() -> LandmarkResult:
    landmarks = [
        Landmark(id=0, x=30, y=40),
        Landmark(id=1, x=44, y=40),
        Landmark(id=2, x=78, y=56),
        Landmark(id=3, x=78, y=76),
        Landmark(id=4, x=48, y=92),
        Landmark(id=5, x=92, y=92),
    ]
    return LandmarkResult(
        model="mediapipe_face_mesh",
        landmark_count=len(landmarks),
        landmarks=landmarks,
        regions={
            "left_eye": [0, 1],
            "right_eye": [0, 1],
            "nose": [2, 3],
            "mouth": [4, 5],
        },
    )


def test_face_usability_gate_returns_no_face_when_bbox_missing():
    gate = FaceUsabilityGate()

    result = gate.evaluate(frame=_face_frame(), face_bbox=None)

    assert result.usable is False
    assert result.no_face is True
    assert result.reason == "no_face_detected"


def test_face_usability_gate_blocks_mouth_cover():
    gate = FaceUsabilityGate()

    result = gate.evaluate(frame=_face_frame(cover_mouth=True), face_bbox=(20, 20, 120, 120))

    assert result.usable is False
    assert result.no_face is True
    assert result.occluded is True
    assert result.reason == "critical_face_region_occluded"
    assert "mouth" in result.occluded_regions or "lower_face" in result.occluded_regions


def test_face_usability_gate_blocks_nose_cover():
    gate = FaceUsabilityGate()

    result = gate.evaluate(frame=_face_frame(cover_nose=True), face_bbox=(20, 20, 120, 120))

    assert result.usable is False
    assert result.reason == "critical_face_region_occluded"
    assert "nose" in result.occluded_regions


def test_face_usability_gate_blocks_eye_cover():
    gate = FaceUsabilityGate()

    result = gate.evaluate(frame=_face_frame(cover_eyes=True), face_bbox=(20, 20, 120, 120))

    assert result.usable is False
    assert result.reason == "critical_face_region_occluded"
    assert "left_eye" in result.occluded_regions or "right_eye" in result.occluded_regions


def test_face_usability_gate_allows_full_visible_face():
    gate = FaceUsabilityGate()

    result = gate.evaluate(frame=_face_frame(), face_bbox=(20, 20, 120, 120))

    assert result.usable is True
    assert result.no_face is False
    assert result.reason == "face_usable"


def test_face_usability_gate_requires_clear_frames_before_recovery():
    gate = FaceUsabilityGate(clear_confirm_frames=3)

    blocked = gate.evaluate(frame=_face_frame(cover_mouth=True), face_bbox=(20, 20, 120, 120))
    recovering_1 = gate.evaluate(frame=_face_frame(), face_bbox=(20, 20, 120, 120))
    recovering_2 = gate.evaluate(frame=_face_frame(), face_bbox=(20, 20, 120, 120))
    recovered = gate.evaluate(frame=_face_frame(), face_bbox=(20, 20, 120, 120))

    assert blocked.usable is False
    assert recovering_1.reason == "recovering_face_usability"
    assert recovering_2.reason == "recovering_face_usability"
    assert recovered.usable is True
    assert recovered.reason == "face_usable"


def test_face_usability_gate_does_not_trust_landmarks_under_occlusion():
    gate = FaceUsabilityGate()

    result = gate.evaluate(
        frame=_face_frame(cover_mouth=True),
        face_bbox=(20, 20, 120, 120),
        landmarks=_hallucinated_landmarks(),
    )

    assert result.usable is False
    assert result.reason == "critical_face_region_occluded"
